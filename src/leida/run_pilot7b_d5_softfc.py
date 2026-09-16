#!/usr/bin/env python3
"""
PILOT 7b (Direction 5 addendum; reliability-only Gate-0 previews + feature-only redundancy).
NO symptom/IQ target is touched. Pre-committed BEFORE running:

  D5B-SOFT  state-weighted FC shift with CONTINUOUS weights (rescue of measured-dead hard gate):
            w_s(t) = <v1_t, c_s>^2  (squared similarity to locked centroid s, sign-invariant)
            dFC_soft_s = blockmeans28(fisherz(weighted_corr(X_half, w_s)) - fisherz(corr(X_half)))
            Every window contributes to every state -> no 8-window binomial collapse.
            Report per-state: mean SB, effective n = (sum w)^2 / sum w^2, and n_eff/T.
  REDUNDANCY (feature-only, no targets): R^2 of each surviving D5 variant's full-run features
            given static FC28 (F_STATIC[:,0:28]) and given static blocks FC28+prec190+phi190+mu190
            (F_STATIC[:,0:598]) — the pilot6 convention for "how much is new vs re-expressed
            static architecture". Linear regression R^2 per component, mean reported.
  ANCHORS   reproduced: occ5 (locked dict), hard-gate dFC_s0 (the measured-dead reference).
"""
import sys, os, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/Volumes/thinkplus/Code/JLucid/src")
import numpy as np
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
NROI = 190
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]

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

def vec_stats(name, F1, F2, sites, mask, results):
    n, d = F1.shape
    per = []
    for j in range(d):
        x1, x2 = F1[mask, j], F2[mask, j]
        ok = np.isfinite(x1) & np.isfinite(x2)
        n_used = int(ok.sum())
        entry = {"comp": j, "n_used": n_used}
        if n_used >= 50 and np.std(x1[ok]) > 1e-12 and np.std(x2[ok]) > 1e-12:
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
        print(f"  {name:16s} d={d:3d} meanSB={results[name]['mean_sb']:.3f} "
              f"siteres={results[name]['mean_sb_siteres']:.3f} >=0.30: {results[name]['n_ge030']}/{len(sbv)}", flush=True)

def redundancy(name, F, FSTAT, blocks, results):
    """Mean per-component R^2 of full-run feature given static blocks (linear regression)."""
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
    sites = np.array(meta["sites"])
    ages = np.array(meta["ages"], float)
    ages = np.nan_to_num(ages, nan=np.nanmean(ages))
    Ts = [t.shape[0] for t in ts_all]
    nS = len(ts_all)
    mask60 = np.array([T >= 60 for T in Ts])

    V1c = np.load(f"{BASE}/results/V1_adhd200_wavelet.npy")
    cent = np.load(f"{BASE}/results/ADHD200_A1_centers.npy")
    yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
    net_idx = [np.array(yeo[n]) for n in NETS]
    labs_all = np.argmax(np.abs(V1c @ cent.T), axis=1).astype(np.int16)
    subs = []; pos = 0
    for T in Ts:
        subs.append((pos, pos + T)); pos += T
    assert pos == V1c.shape[0]
    FSTATIC = np.load(f"{BASE}/results/F_STATIC.npy")

    # raw = pilot7 outputs (instant re-analysis)
    Z = np.load(f"{OUT}/pilot7_d5_raw.npz")
    occ1, occ2, occF = Z["occ1"], Z["occ2"], Z["occF"]
    CSF = Z["CSF"]; DDF = Z["DDF"]; XTMPL = Z["XTMPL"]
    VPHI, VMU = Z["VPHI"], Z["VMU"]

    results = {}
    print("\n=== D5B-SOFT: state-weighted FC shift, continuous weights ===", flush=True)
    neff_all = []
    dFCs1 = np.full((nS, K, 28), np.nan); dFCs2 = np.full((nS, K, 28), np.nan); dFCsF = np.full((nS, K, 28), np.nan)
    t1 = time.time()
    for i in range(nS):
        ts = np.asarray(ts_all[i], float)
        v1 = V1c[subs[i][0]:subs[i][1]]
        T = ts.shape[0]
        for tag, sl, dest in (("h1", slice(0, T // 2), dFCs1), ("h2", slice(T // 2, T), dFCs2),
                              ("full", slice(0, T), dFCsF)):
            X = ts[sl]; W = v1[sl]; Th = X.shape[0]
            if Th < 20:
                continue
            zall = np.arctanh(np.clip(np.nan_to_num(np.corrcoef(X.T)), -0.999999, 0.999999))
            ball_ = blockmeans28(zall, net_idx)
            sims = (W @ cent.T) ** 2  # (Th, K) continuous, sign-invariant
            for s in range(K):
                w = sims[:, s]
                sw = w.sum()
                if sw < 1e-9:
                    continue
                mu = (w @ X) / sw
                Xc = X - mu
                S = (Xc * w[:, None]).T @ Xc / sw
                d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
                C = S / np.outer(d, d)
                z = np.arctanh(np.clip(np.nan_to_num(C), -0.999999, 0.999999))
                dest[i, s] = blockmeans28(z, net_idx) - ball_
                if tag == "full":
                    neff_all.append((sw ** 2) / (np.sum(w * w) + 1e-12) / Th)
        if i % 200 == 0:
            print(f"  subj {i}/{nS} ({time.time()-t1:.0f}s)", flush=True)
    print(f"soft-gate pass done {time.time()-t1:.0f}s", flush=True)
    neff_all = np.array(neff_all).reshape(nS, K)
    results["soft_neff_frac_T"] = {s: float(np.nanmean(neff_all[mask60, s])) for s in range(K)}
    print(f"effective-n fraction of T per state: {[round(v,3) for v in results['soft_neff_frac_T'].values()]}", flush=True)
    for s in range(K):
        vec_stats(f"dFCsoft_s{s}", dFCs1[:, s], dFCs2[:, s], sites, mask60, results)

    print("\n=== anchors (repro from pilot7 raw) ===", flush=True)
    vec_stats("occ5_anchor", occ1, occ2, sites, mask60, results)

    print("\n=== redundancy (feature-only, no targets) ===", flush=True)
    age_z = (ages - ages.mean()) / (ages.std() + 1e-12)
    omF = np.nanmean(occF, 0)
    CxAF = (occF - omF) * age_z[:, None]
    CT14F = np.concatenate([occF @ VPHI, occF @ VMU], axis=1)
    for nm, Fm in [("CxA5_centered", CxAF), ("CS14", CSF), ("T10E5_15", DDF),
                   ("dFCsoft_s0", dFCsF[:, 0]), ("dFCsoft_s2", dFCsF[:, 2]),
                   ("dFCsoft_s4", dFCsF[:, 4]), ("occ5", occF), ("CT14_tmpl", CT14F)]:
        redundancy(nm, Fm, FSTATIC[:, 0:28], "FC28", results)
    for nm, Fm in [("CS14", CSF), ("T10E5_15", DDF)]:
        redundancy(nm, Fm, FSTATIC[:, 0:598], "FC28+prec+phi+mu", results)

    results["wall_secs"] = int(time.time() - t0)
    with open(f"{OUT}/pilot7b_d5_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> pilot7b_d5_results.json", flush=True)
