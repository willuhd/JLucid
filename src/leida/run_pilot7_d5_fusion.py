#!/usr/bin/env python3
"""
PILOT 7 (Direction 5 — FUSION AND INTERACTION; reliability-only Gate-0 previews).
NO symptom/IQ target is ever touched here. Age is used ONLY as a feature multiplier
(maturation interaction), never as a prediction target. Protocol verbatim from pilots 1-6:
split-half = first vs second half of each run; SB = 2r/(1+r) (Pearson across subjects);
n = 871 with T>=60 per exp/37 PREREG; site-residualized alongside raw; ANOVA sigma2_W/sigma2_T.

Pre-committed candidate list (frozen BEFORE running; every point reported, incl. expected-dead):

  D5-A  occ x age interaction (locked exp/21 k=5 dict on cached V1):
        f_s = (occ_s - mean_s) * (age - mean_age), s=0..4  -> 5 comps. Anchor: occ5 itself.
        (per-component affine rescaling does not change split-half r, so SB(f) is identical
         for the z(occ)*z(age) parameterization)
  D5-B  state-conditional FC shift (states as gates):
        per state s with n_s >= 8 windows in that half (locked dict):
          dFC_s = blockmean28( fisherz(corr(BOLD[W_s])) - fisherz(corr(BOLD[all half rows])) )
        -> 5 states x 28 = 140 comps. Anchor: whole-half FC28 (no state conditioning).
        Known risk (pre-committed): n_s~8 windows saturate Fisher-z at the house clip;
        rare states expected dead — reported anyway.
  D5-C  occupancy-weighted controllability:
        C-T (template; direction-literal "per-state precision from that state's windows",
             NOT the subject's own): Theta_s^tmpl = inv(LW(pooled-cohort per-subject-z-scored
             state-s frames, subsample<=40000 rng(0)) + 0.1 I) — FIXED across subjects;
             feature = sum_s occ_s * [netmean7(phi_s^tmpl) || netmean7(mu_s^tmpl)] -> 14 comps.
             By construction a FIXED linear map of occ -> Gate-0 SB = occ's SB.
        C-S (subject; sibling): Theta_s = l2_theta(subject's own state-s rows), n_s >= 10
             else state skipped+renormalized; feature = sum_{s usable} occ_s *
             [netmean7(phi_s) || netmean7(mu_s)] / sum_{s usable} occ_s -> 14 comps.
             Differs from exp/31 (35-dim per-state block, fold-local dict, NaN-impute).
  D5-D  template-state transition energies under full-run architecture:
        Theta = l2_theta(all half rows); A = theta_to_A(Theta); |lam|max guard rescale
        (exp/28 control_energy verbatim); W^{-1} = Q diag(1-lam^2) Q^T (symmetric closed
        form; verified vs scipy bilinear to 5e-10); template targets x_s = unit-norm
        pooled-cohort mean z-BOLD pattern of state-s windows (population constants):
          T10: E_ij = (x_j - x_i)^T W^{-1} (x_j - x_i), 0<=i<j<=4 -> 10 comps
          E5:  e_j  = x_j^T W^{-1} x_j                        -> 5 comps
        Reliability-first: no per-state Gramian (exp/31 C15 conditioned on n_s windows);
        every window contributes to W; targets are noise-free constants.

All exp/28 machinery verbatim: l2_theta (LAM=0.1), theta_to_A, avg_modal, FC28 block convention.
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
from sklearn.covariance import LedoitWolf
from controllability.datasets_cc200 import load_cc200

BASE = "/Volumes/thinkplus/Code/JLucid"
OUT = f"{BASE}/results/pilot_geometry"
TR = 2.0
K = 5
NROI = 190
LAM = 0.1
N_SUB_TMPL = 40000
NETS = ['VIS', 'SOM', 'DAN', 'SAL', 'LIM', 'FPN', 'DMN']
UP = [(a, b) for a in range(7) for b in range(a, 7)]

# ---------------- exp/28 verbatim feature machinery ----------------
def l2_theta(X, lam=LAM):
    Xz = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-12)
    lw = LedoitWolf().fit(Xz)
    return np.linalg.inv(lw.covariance_ + lam * np.eye(X.shape[1]))

def theta_to_A(Theta):
    Nn = Theta.shape[0]
    d = np.sqrt(np.diag(Theta)); d[d == 0] = 1e-12
    P = -Theta / np.outer(d, d); np.fill_diagonal(P, 0)
    P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
    Araw = np.abs(P)
    lmax = float(np.max(np.linalg.eigvalsh((Araw + Araw.T) / 2)))
    A = Araw / (1.0 + lmax + 1e-12) - np.eye(Nn)
    return (A + A.T) / 2

def avg_modal(A):
    vals, vecs = np.linalg.eigh(A)
    assert vals.max() < 0, "A not Hurwitz"
    phi = (vecs ** 2) @ (1.0 / (-2.0 * vals))
    mu = (vecs ** 2) @ (1.0 - np.exp(vals))
    return phi, mu

def winv_from_A(A):
    """exp/28 control_energy |lam| guard, then symmetric closed form:
    A = Q diag(lam) Q^T  =>  A W A^T - W + I = 0  =>  W^{-1} = Q diag(1-lam^2) Q^T."""
    lam_max = float(np.max(np.abs(np.linalg.eigvalsh((A + A.T) / 2))))
    if lam_max >= 1.0:
        A = A / (lam_max + 1e-6) * 0.99
    vals, vecs = np.linalg.eigh((A + A.T) / 2)
    return (vecs * (1.0 - vals ** 2)) @ vecs.T

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

def netmean7(v, net_idx):
    return np.array([v[ix].mean() for ix in net_idx])

# ---------------- stats helpers (verbatim protocol from pilots 1-6) ----------------
def site_resid(X, sites):
    uniq = np.unique(sites)
    D = (sites[:, None] == uniq[None, :]).astype(float)[:, 1:]
    D = np.column_stack([np.ones(len(sites)), D])
    coef, *_ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ coef

def sb(r): return 2 * r / (1 + abs(r)) if r > -1 else -1

# ---------------- globals shared via fork ----------------
NET_IDX = None   # list of 7 index arrays
XTMPL = None     # (5, 190) template state patterns (unit norm)
VPHI = None      # (5, 7) template per-state phi net means
VMU = None       # (5, 7) template per-state mu net means

def worker(args):
    ts, lab = args
    T = ts.shape[0]
    res = {}
    for tag, sl in (("h1", slice(0, T // 2)), ("h2", slice(T // 2, T)), ("full", slice(0, T))):
        X = np.asarray(ts[sl], float)
        l = lab[sl]
        Th = X.shape[0]
        occ = np.bincount(l, minlength=K) / max(Th, 1)
        # ---- D5-B: state-conditional FC shift (n_s >= 8)
        zall = np.arctanh(np.clip(np.nan_to_num(np.corrcoef(X.T)), -0.999999, 0.999999))
        ball = blockmeans28(zall, NET_IDX)
        dFC = np.full((K, 28), np.nan)
        for s in range(K):
            idx = np.flatnonzero(l == s)
            if len(idx) >= 8:
                zs = np.arctanh(np.clip(np.nan_to_num(np.corrcoef(X[idx].T)), -0.999999, 0.999999))
                dFC[s] = blockmeans28(zs, NET_IDX) - ball
        # ---- D5-C (C-S): occ-weighted subject per-state controllability (n_s >= 10)
        w = np.zeros(K); phis = np.zeros((K, 7)); mus = np.zeros((K, 7))
        for s in range(K):
            idx = np.flatnonzero(l == s)
            if len(idx) >= 10:
                try:
                    Theta = l2_theta(X[idx])
                    A = theta_to_A(Theta)
                    phi, mu = avg_modal(A)
                    w[s] = len(idx) / Th
                    phis[s] = netmean7(phi, NET_IDX)
                    mus[s] = netmean7(mu, NET_IDX)
                except Exception:
                    pass
        if w.sum() > 0:
            cs = np.concatenate([(w[:, None] * phis).sum(0) / w.sum(),
                                 (w[:, None] * mus).sum(0) / w.sum()])
        else:
            cs = np.full(14, np.nan)
        # ---- D5-D: template-state transition energies under full-slice architecture
        try:
            Theta = l2_theta(X)
            A = theta_to_A(Theta)
            Winv = winv_from_A(A)
            T10 = []; E5 = []
            for i in range(K):
                E5.append(float(XTMPL[i] @ Winv @ XTMPL[i]))
                for j in range(i + 1, K):
                    d = XTMPL[j] - XTMPL[i]
                    T10.append(float(d @ Winv @ d))
            dd = np.concatenate([np.array(T10), np.array(E5)])
        except Exception:
            dd = np.full(15, np.nan)
        res[tag] = {"occ": occ, "dFC": dFC, "CS": cs, "DD": dd, "ball": ball,
                    "nstate": np.array([(l == s).sum() for s in range(K)])}
    return res

def vec_stats(name, F1, F2, Ff, sites, mask, results, group_of=None):
    """Per-component split-half r/SB on the T>=60 mask; NaN-aware; group means optional."""
    n, d = F1.shape
    per = []
    for j in range(d):
        x1, x2, xf = F1[mask, j], F2[mask, j], Ff[mask, j]
        ok = np.isfinite(x1) & np.isfinite(x2)
        n_used = int(ok.sum())
        entry = {"comp": j, "n_used": n_used}
        if n_used >= 50 and np.std(x1[ok]) > 1e-12 and np.std(x2[ok]) > 1e-12:
            sv = sites[mask][ok]
            r = float(np.corrcoef(x1[ok], x2[ok])[0, 1])
            rr = float(np.corrcoef(site_resid(x1[ok], sv), site_resid(x2[ok], sv))[0, 1])
            dd_ = x1[ok] - x2[ok]
            sw2 = float(np.mean(dd_ * dd_) / 2.0)
            of = np.isfinite(xf)
            st2 = max(float(np.var(xf[of])) - sw2 / 2.0, 1e-12) if of.sum() >= 50 else None
            entry.update({"r": r, "sb": sb(r), "sb_siteres": sb(rr), "sigma2_W": sw2})
            if st2 is not None:
                entry.update({"sigma2_T": st2, "r_pred_anova": st2 / (st2 + sw2)})
        per.append(entry)
    sbv = [p["sb"] for p in per if "sb" in p]
    if sbv:
        results[name] = {"n_comp": len(sbv), "comps": d,
                        "mean_sb": float(np.mean(sbv)),
                        "mean_sb_siteres": float(np.mean([p["sb_siteres"] for p in per if "sb_siteres" in p])),
                        "n_ge030": int(sum(1 for v in sbv if v >= 0.30)),
                        "per_comp": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items()} for p in per]}
        print(f"  {name:12s} d={d:3d} meanSB={results[name]['mean_sb']:.3f} "
              f"siteres={results[name]['mean_sb_siteres']:.3f} >=0.30: {results[name]['n_ge030']}/{len(sbv)}", flush=True)
    else:
        print(f"  {name:12s} d={d:3d} NO USABLE COMPONENTS", flush=True)
        results[name] = {"n_comp": 0, "comps": d, "per_comp": per}

if __name__ == "__main__":
    t0 = time.time()
    print("loading cc200 ...", flush=True)
    ts_all, labels_all, meta = load_cc200(qc_only=True)
    sites = np.array(meta["sites"])
    ages = np.array(meta["ages"], float)
    Ts = [t.shape[0] for t in ts_all]
    print(f"loaded {len(ts_all)} in {time.time()-t0:.0f}s total_T={sum(Ts)}", flush=True)

    V1c = np.load(f"{BASE}/results/V1_adhd200_wavelet.npy")
    cent = np.load(f"{BASE}/results/ADHD200_A1_centers.npy")
    yeo = json.load(open(f"{BASE}/src/leida/atlas/yeo_true_idx.json"))
    NET_IDX = [np.array(yeo[n]) for n in NETS]
    print(f"V1 {V1c.shape} cent {cent.shape}", flush=True)

    # locked-dict labels (verbatim pilot6 repro)
    labs_all = np.argmax(np.abs(V1c @ cent.T), axis=1).astype(np.int16)
    subs = []; pos = 0
    for T in Ts:
        subs.append((pos, pos + T)); pos += T
    assert pos == V1c.shape[0]
    labs_subj = [labs_all[a:b] for a, b in subs]

    # cohort descriptors (no targets)
    mask60 = np.array([T >= 60 for T in Ts])
    print(f"T>=60: {mask60.sum()}/{len(Ts)} | T min/med/max {min(Ts)}/{int(np.median(Ts))}/{max(Ts)}", flush=True)
    ages_c = ages.copy()
    n_age_nan = int(np.isnan(ages_c).sum())
    ages_c[np.isnan(ages_c)] = np.nanmean(ages_c)
    age_z = (ages_c - ages_c.mean()) / (ages_c.std() + 1e-12)
    print(f"age: nan={n_age_nan} mean={ages_c.mean():.2f} sd={ages_c.std():.2f} range {ages_c.min():.1f}-{ages_c.max():.1f}", flush=True)
    occ_full_all = np.array([np.bincount(l, minlength=K) / max(len(l), 1) for l in labs_subj])
    print(f"mean occ per state (locked dict): {occ_full_all.mean(0).round(3).tolist()}", flush=True)
    h1n = np.array([np.bincount(l[:len(l)//2], minlength=K) for l in labs_subj])
    h2n = np.array([np.bincount(l[len(l)//2:], minlength=K) for l in labs_subj])
    print(f"usable n_s>=8 both halves:  {[( (h1n>=8)&(h2n>=8) )[mask60, s].mean().round(3) for s in range(K)]}", flush=True)
    print(f"usable n_s>=10 both halves: {[( (h1n>=10)&(h2n>=10) )[mask60, s].mean().round(3) for s in range(K)]}", flush=True)

    # ---------------- template pass (C-T constants + D5-D targets) ----------------
    print("\ntemplate pass (pooled per-state z-BOLD frames) ...", flush=True)
    t1 = time.time()
    state_rows = {s: [] for s in range(K)}
    for i in range(len(ts_all)):
        X = np.asarray(ts_all[i], float)
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
        for s in range(K):
            idx = np.flatnonzero(labs_subj[i] == s)
            if len(idx):
                state_rows[s].append(Xz[idx])
    XTMPL = np.zeros((K, NROI)); VPHI = np.zeros((K, 7)); VMU = np.zeros((K, 7))
    rng = np.random.default_rng(0)
    for s in range(K):
        A = np.concatenate(state_rows[s], 0)
        n_pool = A.shape[0]
        if n_pool > N_SUB_TMPL:
            A = A[rng.choice(n_pool, N_SUB_TMPL, replace=False)]
        xs = A.mean(0); xs = xs / (np.linalg.norm(xs) + 1e-12)
        XTMPL[s] = xs
        Theta = np.linalg.inv(LedoitWolf().fit(A).covariance_ + LAM * np.eye(NROI))
        As = theta_to_A(Theta)
        phi, mu = avg_modal(As)
        VPHI[s] = netmean7(phi, NET_IDX); VMU[s] = netmean7(mu, NET_IDX)
        print(f"  state {s}: pooled {n_pool} frames, |x|={np.linalg.norm(A.mean(0)):.3f}", flush=True)
    del state_rows
    print(f"template pass done in {time.time()-t1:.0f}s", flush=True)

    # ---------------- per-subject pass ----------------
    print("\nper-subject pass (serial in-process; mp.Pool fork hangs in this sandbox) ...", flush=True)
    t1 = time.time()
    outs = []
    for i in range(len(ts_all)):
        outs.append(worker((ts_all[i], labs_subj[i])))
        if i % 100 == 0:
            print(f"  subj {i}/{len(ts_all)} ({time.time()-t1:.0f}s)", flush=True)
    print(f"per-subject pass done in {time.time()-t1:.0f}s", flush=True)

    nS = len(ts_all)
    def stack(key, tag, shape):
        out = np.full((nS,) + shape, np.nan)
        for i, r in enumerate(outs):
            out[i] = r[tag][key]
        return out
    # persist raw worker outputs for instant re-analysis
    np.savez_compressed(f"{OUT}/pilot7_d5_raw.npz",
            occ1=stack("occ", "h1", (K,)), occ2=stack("occ", "h2", (K,)), occF=stack("occ", "full", (K,)),
            dFC1=stack("dFC", "h1", (K, 28)), dFC2=stack("dFC", "h2", (K, 28)), dFCF=stack("dFC", "full", (K, 28)),
            CS1=stack("CS", "h1", (14,)), CS2=stack("CS", "h2", (14,)), CSF=stack("CS", "full", (14,)),
            DD1=stack("DD", "h1", (15,)), DD2=stack("DD", "h2", (15,)), DDF=stack("DD", "full", (15,)),
            ball1=stack("ball", "h1", (28,)), ball2=stack("ball", "h2", (28,)), ballF=stack("ball", "full", (28,)),
            XTMPL=XTMPL, VPHI=VPHI, VMU=VMU, sites=sites, ages=ages, Ts=np.array(Ts))

    occ1 = stack("occ", "h1", (K,)); occ2 = stack("occ", "h2", (K,)); occF = stack("occ", "full", (K,))
    dFC1 = stack("dFC", "h1", (K, 28)); dFC2 = stack("dFC", "h2", (K, 28)); dFCF = stack("dFC", "full", (K, 28))
    CS1 = stack("CS", "h1", (14,)); CS2 = stack("CS", "h2", (14,)); CSF = stack("CS", "full", (14,))
    DD1 = stack("DD", "h1", (15,)); DD2 = stack("DD", "h2", (15,)); DDF = stack("DD", "full", (15,))
    ball1 = stack("ball", "h1", (28,)); ball2 = stack("ball", "h2", (28,)); ballF = stack("ball", "full", (28,))

    results = {}
    print("\n=== Gate-0 previews (n = T>=60 subjects) ===", flush=True)

    # anchor: locked-dict occupancy (reproduces pilot6 ANCHOR_occ5)
    vec_stats("occ5_anchor", occ1, occ2, occF, sites, mask60, results)

    # D5-A: occ x age interaction
    AxA1 = occ1 * age_z[:, None]; AxA2 = occ2 * age_z[:, None]; AxAF = occF * age_z[:, None]
    vec_stats("AxA5_D5A", AxA1, AxA2, AxAF, sites, mask60, results)
    # D5-A companion (audit item): centered interaction — subject-mean-center occ BEFORE
    # multiplying, removing the mean_occ x age main-effect term; the raw product's SB is
    # inflated by the trivially-reliable level x age component.
    omF = np.nanmean(occF, 0)
    CxA1 = (occ1 - omF) * age_z[:, None]; CxA2 = (occ2 - omF) * age_z[:, None]
    CxAF = (occF - omF) * age_z[:, None]
    vec_stats("CxA5_D5A_centered", CxA1, CxA2, CxAF, sites, mask60, results)

    # anchor: whole-half FC28 (no state conditioning)
    vec_stats("FC28_anchor", ball1, ball2, ballF, sites, mask60, results)

    # D5-B: state-conditional FC shift, per state
    for s in range(K):
        vec_stats(f"dFC_s{s}_D5B", dFC1[:, s], dFC2[:, s], dFCF[:, s], sites, mask60, results)

    # D5-C template (occ-weighted fixed per-state controllability) = fixed linear map of occ
    CT1 = np.concatenate([occ1 @ VPHI, occ1 @ VMU], axis=1)
    CT2 = np.concatenate([occ2 @ VPHI, occ2 @ VMU], axis=1)
    CTF = np.concatenate([occF @ VPHI, occF @ VMU], axis=1)
    vec_stats("CT14_D5Ctmpl", CT1, CT2, CTF, sites, mask60, results)

    # D5-C subject (occ-weighted own-window per-state controllability, skip+renorm)
    vec_stats("CS14_D5Csubj", CS1, CS2, CSF, sites, mask60, results)

    # D5-D: transition energies T10 + entry energies E5
    vec_stats("T10_D5Dtrans", DD1[:, :10], DD2[:, :10], DDF[:, :10], sites, mask60, results)
    vec_stats("E5_D5Dentry", DD1[:, 10:], DD2[:, 10:], DDF[:, 10:], sites, mask60, results)

    # usability fractions for C-S (any usable state)
    cs_ok1 = np.isfinite(CS1).all(1); cs_ok2 = np.isfinite(CS2).all(1)
    results["usability"] = {
        "ns_ge8_both_frac": [float(((h1n >= 8) & (h2n >= 8))[mask60, s].mean()) for s in range(K)],
        "ns_ge10_both_frac": [float(((h1n >= 10) & (h2n >= 10))[mask60, s].mean()) for s in range(K)],
        "CS_any_state_usable_frac": float((cs_ok1 & cs_ok2)[mask60].mean()),
        "mean_occ": occ_full_all.mean(0).tolist(),
        "age_mean": float(ages_c.mean()), "age_sd": float(ages_c.std()),
        "age_nan": n_age_nan, "n_T60": int(mask60.sum()),
        "wall_secs": int(time.time() - t0),
    }
    with open(f"{OUT}/pilot7_d5_results.json", "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwall {time.time()-t0:.0f}s -> pilot7_d5_results.json", flush=True)
